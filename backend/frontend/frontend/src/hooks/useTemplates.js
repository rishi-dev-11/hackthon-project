import { useState, useEffect } from 'react';
import { getTemplates, getTemplate } from '../services/api';

export const useTemplates = () => {
  const [templates, setTemplates] = useState([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);

  useEffect(() => {
    fetchTemplates();
  }, []);

  const fetchTemplates = async () => {
    try {
      setLoading(true);
      const data = await getTemplates();
      setTemplates(data);
      setError(null);
    } catch (err) {
      setError(err.message);
      console.error('Error fetching templates:', err);
    } finally {
      setLoading(false);
    }
  };

  const fetchTemplate = async (templateId) => {
    try {
      setLoading(true);
      const data = await getTemplate(templateId);
      return data;
    } catch (err) {
      setError(err.message);
      console.error('Error fetching template:', err);
      throw err;
    } finally {
      setLoading(false);
    }
  };

  return {
    templates,
    loading,
    error,
    refreshTemplates: fetchTemplates,
    getTemplateById: fetchTemplate,
  };
};
